Replication Package
===================

Title:
Early Warning Systems for Systemic Banking Distress: Persistence, Covariates, and Predictive Performance in Small Macroprudential Panels

Author:
Hedson Malata

Purpose:
This replication package reproduces the empirical results, robustness checks, diagnostic tables, and figures reported in the manuscript.

Files:
1. cleaned_full_panel.csv
   Cleaned macroprudential panel used for estimation. The dataset includes country-year identifiers, prudential indicators, constructed distress indicators, lagged predictors, and persistence variables.

2. ews_replication_script_master.py
   Main Python replication script. Running this script generates the model outputs, diagnostics, robustness checks, and figures.

3. ews_replication_script_master.ipynb
   Notebook version of the replication script for interactive inspection.

4. requirements.txt
   List of required Python packages.

5. outputs/
   Folder containing generated tables and prediction files.

6. figures/
   Folder containing generated figures.

Software Requirements:
Python 3.10 or later is recommended.

Required packages:
numpy
pandas
matplotlib
seaborn
scikit-learn
scipy
statsmodels
openpyxl

How to Run:
Place the following files in the same folder:

cleaned_full_panel.csv
ews_replication_script_master.py
requirements.txt

Then run:

python ews_replication_script_master.py

Alternatively, the notebook version can be opened and run in Jupyter.

Main Outputs:
The script generates the following outputs:

outputs/results_table_main.csv
outputs/bootstrap_auc_comparisons.csv
outputs/bootstrap_auc_mean_differences.csv
outputs/correlation_matrix.csv
outputs/vif_diagnostics.csv
outputs/logit_coefficient_summary.csv
outputs/woe_robustness_results.csv
outputs/woe_coefficient_summary.csv
outputs/oos_predictions_hybrid.csv
outputs/oos_predictions_with_woe_logit.csv

Main Figures:
The script generates the following figures:

figures/roc_curves.png
figures/precision_recall_curves.png
figures/probability_distributions.png
figures/appendix_calibration_curve.png
figures/country_probability_trajectories.png
figures/appendix_pooled_probability_signals.png

Notes:
The panel is unbalanced. Missing values reflect unavailable observations in the original macroprudential indicators and lag construction for recursive forecasting. The released dataset is preserved as used in the estimation. Model-specific handling of missing values is implemented within the replication script.

The primary target variable is y_true. The alternative broader fragility target is y_alt. The target can be changed in the script by setting:

TARGET_COL = "y_true"

or

TARGET_COL = "y_alt"

Random Seed:
The script uses RANDOM_STATE = 42 for reproducibility.

Contact:
Hedson Malata
