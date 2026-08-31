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

4. requirements.txt

   List of Python packages required to run the replication code.

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
