
## Replication Repository

This repository contains the replication materials for the paper:

“Early Warning Systems for Systemic Banking Distress: Persistence, Covariates, and Predictive Performance in Small Macroprudential Panels”** by Hedson Malata, accepted for publication in _Finance Research Letters_.

The repository provides the analysis-ready country-year panel dataset, reproducible Python source code, out-of-sample prediction files, diagnostic outputs, and supporting tables required to reproduce and verify the empirical results reported in the paper.

The systemic-distress measure is constructed from four binary indicators representing capital weakness, profitability deterioration, asset-quality weakness, and liquidity vulnerability:

* capital fragility: \(EAR<0.04\);
* profitability fragility: \(ROA\leq0\);
* asset-quality fragility: \(LLR>0.03\); and
* liquidity fragility: \(LIQ<0.20\) or \(WFS>0.20\).

The composite stress score is the sum of these four indicators. The baseline severe-distress outcome equals one when at least two component conditions are triggered. When an underlying indicator is unavailable, that indicator does not independently trigger its corresponding threshold condition.

Numerical zeros recorded in the analysis dataset are retained as observed or calculated values and are distinct from unavailable observations, which remain coded as missing. In particular, no observed LLR value exceeds the 3% asset-quality threshold; consequently, the asset-quality component does not independently trigger a distress event in the analysis panel.

Applying the stated definitions to the supplied dataset reproduces all four component indicators, the composite stress score, and the baseline distress outcome exactly for all 125 country-year observations.

The repository reproduces the recursive out-of-sample forecasting analysis, model comparisons, robustness exercises, diagnostic statistics, tables, and figures reported in the paper. The reported 107 observations constitute the recursive out-of-sample forecast-evaluation panel following the three-year country-specific warm-up period. This figure should not be interpreted as the number of observations with complete values for every continuous prudential indicator. Effective estimation samples vary across recursive windows according to predictor availability.

The persistence-based structural model uses historical distress realizations and therefore does not directly require complete EAR, ROA, LLR, or LIQ observations. For the ridge-logit specification, training observations with incomplete lagged predictors are excluded. When a lagged predictor is unavailable for an individual forecast observation, the missing value is handled by the median-imputation step contained in the fitted model pipeline.


Repository Contents
-------------------

1. analysis_dataset.csv

   Analysis-ready panel dataset used for model estimation and forecast
   evaluation. The dataset contains country and year identifiers, prudential
   indicators, constructed banking distress measures, lagged explanatory
   variables, persistence variables, and macro-financial covariates used in
   the forecasting models.

2. EWS_reproducible_Source_Codes.py

   Main Python replication script. Running this script reproduces the complete
   empirical analysis, including data preprocessing, model estimation,
   recursive out-of-sample forecasting, resampling and bootstrap analyses,
   robustness checks, diagnostic statistics, and the tables and prediction
   outputs reported in the manuscript.

3. EWS_reproducible_Source_Codes.ipynb

   Interactive Jupyter Notebook version of the replication script for
   inspection, exploratory analysis, and step-by-step reproduction of the
   empirical results.

4. Software Requirements

   The replication code requires Python 3.10 or later and the following
   Python packages:

   - numpy
   - pandas
   - matplotlib
   - seaborn
   - scikit-learn
   - scipy
   - statsmodels
   - openpyxl

5. outputs/

   Directory containing the generated manuscript tables, prediction files,
   diagnostic results, and supporting replication outputs.


Main Outputs
------------

Running the replication script generates the following outputs in the
`outputs/` directory.


Main Manuscript Tables

- table_1_out_of_sample_model_performance.csv
- table_2_distress_environments.csv
- table_3_calibration_diagnostics.csv
- table_4_recursive_forecasting_stability.csv
- table_5_woe_robustness_main.csv
- table_6_bootstrap_auc_differences_main.csv


Appendix Tables

- table_A1_recursive_logit_coefficients.csv
- table_A2_woe_coefficient_diagnostics_display.csv
- table_A3_descriptive_statistics.csv
- table_A4_country_level_distress_frequencies.csv
- table_A5_oos_severe_distress_distribution.csv
- table_A6_persistence_overlap_diagnostic.csv
- table_A7_leave_one_country_out_influence.csv
- table_A8_threshold_sensitivity.csv
- table_A9_macro_financial_proxy_augmentation.csv
- table_A10_kappa_sensitivity.csv


Diagnostic and Supporting Outputs

- appendix_recursive_logit_coefficients.csv
- correlation_matrix.csv
- macro_augmented_model_results.csv
- vif_diagnostics.csv
- woe_coefficient_diagnostics.csv
- woe_coefficient_draws.csv
- woe_logit_coefficients_main.csv


Prediction Files

- oos_predictions_main_with_woe.csv
- oos_predictions_strict_distress.csv
- oos_predictions_macro_augmented.csv
