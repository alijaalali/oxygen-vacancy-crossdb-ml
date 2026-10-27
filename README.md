# Cross-database ML for oxygen-vacancy formation energy (Emery & Wolverton vs. Witman/NREL/Sandia/LLNL vs. JARVIS-DFT)

Scripts used to build features, train models, run the robustness/sensitivity analyses, and generate figures/tables for the manuscript.

## Pipeline order

1. `cross_database_pipeline.py` — stability filter, Magpie featurization, zero-shot cross-database transfer, uncertainty quantification, active-learning simulation
2. `make_predictions.py` — regenerates per-compound predictions and UQ ensemble output from the fitted pipeline
3. `label_efficiency_and_redox_diagnosis.py` — label-efficiency (joint-training) curve and redox-active-cation transfer-error diagnosis
4. `multimodel_robustness_check.py` — repeats the zero-shot transfer test, redox diagnosis, and label-efficiency curve across 10 regressors
5. `duplicate_and_distance_robustness_checks.py` — duplicate-composition reweighting and feature-space distance-to-training checks
6. `sensitivity_and_bootstrap_checks.py` — stability-threshold sensitivity sweep, size-matched subgroup control, composition-overlap check, bootstrap CIs
7. `calibration_subset_offset_correction.py` — offset correction estimated from a held-out calibration subset only
8. `hyperparameter_search.py` — 5-fold CV hyperparameter sweep
9. `extended_robustness_analyses.py` — structural-family-matched redox comparison, both label-efficiency split protocols, direct formula comparison, intra-compound site variation, rank correlation, UQ-vs-distance baseline, multiseed active learning
10. `target_magnitude_control.py` — matched-subset regression controlling for redox activity, structural family, feature-space distance, and target-energy magnitude
11. `negative_transfer_controls.py` — source-downsampling and target-reweighting controls for the negative-transfer finding
12. `composition_averaged_check.py` — re-runs the zero-shot and joint-training comparison at the composition-averaged (not site-level) resolution
13. `subgroup_statistics.py` — MAE/RMSE/target distribution and composition-grouped bootstrap CI for the redox-active vs. fixed-valence subgroup comparison
14. `active_learning_variability.py` — across-seed variability of the active-learning labels-to-threshold comparison
15. `ridge_baseline.py` — ridge-regression and mean-prediction baselines on the same in-distribution split, for comparison against the primary gradient-boosted-tree model
16. `k_composition_counts.py` — mean number of unique compositions represented in the label-efficiency pool at each label budget k

## Figures

- `make_figures.py` — main manuscript figures
- `make_figure_label_efficiency_comparison.py` — composition-grouped vs. entry-random label-efficiency curves
- `make_figures_label_efficiency_redox.py` — label-efficiency curve and redox diagnosis figures
- `make_figures_multimodel.py` — multi-model robustness figures
- `make_figures_robustness.py` — duplicate-composition and feature-distance robustness figures
- `make_figure_uq_redox_stratified.py` — uncertainty-vs-error scatter stratified by redox class

## Requirements

Python 3, pandas, numpy, scikit-learn, scipy, pymatgen, matminer, matplotlib, joblib.

## Data

Raw data (Emery & Wolverton oxide screen, Witman/NREL/Sandia/LLNL defect database, JARVIS-DFT vacancydb) are not included here; scripts expect them in a local `data/` directory. See the manuscript's Data Availability statement.
