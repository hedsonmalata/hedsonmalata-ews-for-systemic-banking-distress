Replication Package
===================

Title
-----
Early Warning Systems for Systemic Banking Distress: Persistence, Covariates, and Predictive Performance in Small Macroprudential Panels

Author
------
Hedson Malata

Overview
--------
This repository contains the replication materials for the study "Early Warning Systems for Systemic Banking Distress: Persistence, Covariates, and Predictive Performance in Small Macroprudential Panels." The package enables users to reproduce the empirical analyses, robustness checks, diagnostic tests, forecasting results, tables, and figures reported in the manuscript.

Repository Contents
-------------------

1. analysis_dataset.csv

   Analysis-ready panel dataset used for model estimation and forecast evaluation. The dataset contains country and year identifiers, prudential indicators, constructed banking distress measures, lagged explanatory variables, persistence variables, and macro-financial covariates used in the forecasting models.

2. EWS_reproducible_Source_Codes.py

   Main Python replication script. Running this script reproduces the complete empirical analysis, including data preprocessing, model estimation, recursive out-of-sample forecasting, bootstrap analyses, robustness checks, diagnostic statistics, tables, and figures reported in the manuscript.

3. EWS_reproducible_Source_Codes.ipynb

   Interactive Jupyter Notebook version of the replication script for exploratory analysis and inspection.

4. requirements.txt

   List of required Python packages.

5. outputs/

   Directory containing generated tables, prediction files, and intermediate outputs.

Software Requirements
---------------------

Python 3.10 or later is recommended.

Required packages:

- numpy
- pandas
- matplotlib
- seaborn
- scikit-learn
- scipy
- statsmodels
- openpyxl

Install the required packages using:

- pip install -r requirements.txt

Running the Replication
-----------------------

Ensure that the following files are located in the same project directory:

- analysis_dataset.csv
- EWS_reproducible_Source_Codes.py
- requirements.txt

Run the replication using:

- python EWS_reproducible_Source_Codes.py

Alternatively, open EWS_reproducible_Source_Codes.ipynb in Jupyter Notebook or JupyterLab and execute all cells sequentially.

Main Outputs
------------

Running the replication script generates the following outputs in the `outputs/` directory, including:

Tables

- table_2_distress_environments.csv
- table_3_calibration_diagnostics.csv
- table_A1_recursive_logit_coefficients.csv
- table_A3_descriptive_statistics.csv
- table_A4_country_level_distress_frequencies.csv
- table_A5_oos_severe_distress_distribution.csv
- table_A8_macro_financial_proxy_augmentation.csv
- table_excluding_south_africa.csv
- table_resampling_stability_main.csv
- table_threshold_sensitivity.csv
- table_woe_robustness_main.csv

Diagnostic Outputs

correlation_matrix.csv
- vif_diagnostics.csv
- bootstrap_auc_differences_main.csv
- macro_augmented_model_results.csv
- appendix_recursive_logit_coefficients.csv
- appendix_leave_one_country_out_influence.csv
- appendix_persistence_overlap_diagnostic.csv
- woe_coefficient_diagnostics.csv
- woe_coefficient_diagnostics_display.csv
- woe_logit_coefficients_main.csv

Prediction Files

- oos_predictions_strict_distress.csv
- oos_predictions_main_with_woe.csv
- oos_predictions_macro_augmented.csv
- oos_predictions_excluding_south_africa.csv
