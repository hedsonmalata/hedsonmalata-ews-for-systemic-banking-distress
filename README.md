## Replication Repository

This repository contains the replication materials for the paper:

**“Early Warning Systems for Systemic Banking Distress: Persistence, Covariates, and Predictive Performance in Small Macroprudential Panels”** by Hedson Malata, accepted for publication in *Finance Research Letters*.

The repository provides the analysis-ready country-year panel dataset, reproducible Python source code, out-of-sample prediction files, diagnostic outputs, and supporting tables required to reproduce and verify the empirical results reported in the paper.

The analysis dataset contains 125 annual country-year observations covering Botswana, Malawi, Namibia, South Africa, Tanzania, and Zambia over the period 2004–2024, subject to source availability. The underlying financial information was manually extracted and harmonised from publicly available annual reports and, where applicable, financial stability reports issued by the respective national central banks. The repository begins with the resulting analysis-ready panel and reproduces the forecasting analysis; it does not reproduce the original manual extraction of figures from those source reports.

The systemic-distress measure is constructed from four binary indicators representing capital weakness, profitability deterioration, asset-quality weakness, and liquidity vulnerability:

* capital fragility: equity-to-assets ratio (\(EAR<0.04\));
* profitability fragility: return on assets (\(ROA\leq0\));
* asset-quality fragility: loan-loss ratio (\(LLR>0.03\)); and
* liquidity fragility: liquid assets to total assets (\(LIQ<0.20\)) or wholesale funding exceeding 20% of total funding (\(WFS>0.20\)).

The composite stress score is the sum of these four indicators. The baseline severe-distress outcome equals one when at least two component conditions are triggered. When an underlying indicator is unavailable, its corresponding threshold comparison does not independently trigger a fragility condition. For the liquidity component, however, either available condition; low LIQ or high WFS, is sufficient to trigger liquidity fragility.

WFS is used in constructing the liquidity-fragility component but is not included among the four continuous lagged predictors in the baseline ridge-logit specification.

Numerical zeros recorded in the analysis dataset are retained as numerical observations and are distinct from unavailable observations, which remain coded as missing. In particular, no observed LLR value exceeds the 3% asset-quality threshold; consequently, the asset-quality component does not independently trigger a distress event in the analysis panel.

Applying the stated definitions to the supplied dataset reproduces all four component indicators, the composite stress score, and the baseline distress outcome exactly for all 125 country-year observations.

In addition to the baseline severe-distress outcome (\(S_{it}\geq2\)), the analysis considers early fragility (\(S_{it}\geq1\)), a stricter distress definition (\(S_{it}\geq3\)), and an onset-focused specification that excludes continuing severe-distress episodes.

The repository reproduces the recursive out-of-sample forecasting analysis, model comparisons, robustness exercises, diagnostic statistics, and numerical results reported in the paper. The reported 107 observations constitute the recursive out-of-sample forecast-evaluation panel following the three-year country-specific warm-up period. This figure should not be interpreted as the number of observations with complete values for every continuous prudential indicator. Effective estimation samples vary across recursive windows according to predictor availability.

The persistence-based structural model uses historical distress realizations and therefore does not directly require complete EAR, ROA, LLR, or LIQ observations. For the ridge-logit specification, training observations with incomplete lagged predictors are excluded. When a lagged predictor is unavailable for an individual forecast observation, it is imputed using the median estimated from the corresponding historical training window, thereby preserving the recursive information structure.

## Repository Contents

1. `analysis_dataset.csv`

   Analysis-ready panel dataset used for model estimation and forecast evaluation. The dataset contains country and year identifiers, prudential indicators, constructed banking-distress measures, lagged explanatory variables, persistence variables, and macro-financial covariates used in the forecasting models.

2. `EWS_reproducible_Source_Codes.py`

   Main Python replication script. Running this script validates and prepares the analysis-ready dataset and reproduces the empirical analysis, including model estimation, recursive out-of-sample forecasting, resampling and bootstrap analyses, robustness checks, diagnostic statistics, and the tables, figures, and prediction outputs reported in the manuscript.

3. `EWS_reproducible_Source_Codes.ipynb`

   Interactive Jupyter Notebook version of the replication script for inspection and step-by-step reproduction of the empirical results.

4. Software requirements

   The replication code requires Python 3.10 or later and the following Python packages:

   * numpy
   * pandas
   * matplotlib
   * seaborn
   * scikit-learn
   * scipy
   * statsmodels
   * openpyxl

5. `outputs/`

   Directory containing the generated manuscript tables, prediction files, diagnostic results, and supporting replication outputs.

## Main Outputs

Running the replication script generates the following outputs in the `outputs/` directory.

### Main Manuscript Tables

* `table_1_out_of_sample_model_performance.csv`
* `table_2_distress_environments.csv`
* `table_3_calibration_diagnostics.csv`
* `table_4_recursive_forecasting_stability.csv`
* `table_5_woe_robustness_main.csv`
* `table_6_bootstrap_auc_differences_main.csv`

### Appendix Tables

* `table_A1_recursive_logit_coefficients.csv`
* `table_A2_woe_coefficient_diagnostics_display.csv`
* `table_A3_descriptive_statistics.csv`
* `table_A4_country_level_distress_frequencies.csv`
* `table_A5_oos_severe_distress_distribution.csv`
* `table_A6_persistence_overlap_diagnostic.csv`
* `table_A7_leave_one_country_out_influence.csv`
* `table_A8_threshold_sensitivity.csv`
* `table_A9_macro_financial_proxy_augmentation.csv`
* `table_A10_kappa_sensitivity.csv`

### Diagnostic and Supporting Outputs

* `appendix_recursive_logit_coefficients.csv`
* `correlation_matrix.csv`
* `macro_augmented_model_results.csv`
* `vif_diagnostics.csv`
* `woe_coefficient_diagnostics.csv`
* `woe_coefficient_draws.csv`
* `woe_logit_coefficients_main.csv`

### Prediction Files

* `oos_predictions_main_with_woe.csv`
* `oos_predictions_strict_distress.csv`
* `oos_predictions_macro_augmented.csv`
