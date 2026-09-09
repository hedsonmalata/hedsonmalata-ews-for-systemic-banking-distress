# Replication Repository

This repository contains the replication materials for the paper:

**“Early Warning Systems for Systemic Banking Distress: Persistence, Covariates, and Predictive Performance in Small Macroprudential Panels”**  
by **Hedson Malata**, accepted for publication in *Finance Research Letters*.

The repository provides the analysis-ready country-year panel dataset, reproducible Python source code, out-of-sample prediction files, diagnostic outputs, and supporting tables required to reproduce and verify the empirical results reported in the paper.

## Data and Replication Scope

The analysis dataset contains 125 annual country-year observations covering Botswana, Malawi, Namibia, South Africa, Tanzania, and Zambia over the period 2004–2024, subject to source availability.

The underlying financial information was manually extracted and harmonized from publicly available annual reports and, where applicable, financial stability reports issued by the respective national central banks. The repository begins with the resulting analysis-ready panel and reproduces the forecasting analysis. It does not reproduce the original manual extraction of figures from those source reports.

The systemic-distress measure is constructed from four binary indicators representing capital weakness, profitability deterioration, asset-quality weakness, and liquidity vulnerability:

* capital fragility: equity-to-assets ratio (\(EAR<0.04\));
* profitability fragility: return on assets (\(ROA\leq0\));
* asset-quality fragility: loan-loss ratio (\(LLR>0.03\)); and
* liquidity fragility: liquid assets to total assets (\(LIQ<0.20\)) or wholesale funding exceeding 20% of total funding (\(WFS>0.20\)).

The composite stress score is

\[
S_{it}
=
D^{(1)}_{it}
+
D^{(2)}_{it}
+
D^{(3)}_{it}
+
D^{(4)}_{it}.
\]

The baseline severe-distress outcome equals one when at least two component conditions are triggered:

\[
y_{it}=\mathbf{1}\{S_{it}\geq2\}.
\]

When an underlying indicator is unavailable, its corresponding threshold comparison does not independently trigger a fragility condition. For the liquidity component, either available condition—low LIQ or high WFS—is sufficient to trigger liquidity fragility.

WFS is used in constructing the liquidity-fragility component but is not included among the four continuous lagged predictors in the baseline ridge-logit specification.

Numerical zeros recorded in the analysis dataset are retained as numerical observations and are distinct from unavailable observations, which remain coded as missing. In particular, no observed LLR value exceeds the 3% asset-quality threshold; consequently, the asset-quality component does not independently trigger a distress condition in the analysis panel.

Applying the stated definitions to the supplied analysis-ready dataset reproduces the four component indicators, the composite stress score, and the baseline distress outcome for all 125 country-year observations.

In addition to the baseline severe-distress outcome (\(S_{it}\geq2\)), the analysis considers:

* early fragility (\(S_{it}\geq1\));
* a stricter distress definition (\(S_{it}\geq3\)); and
* a purified onset-focused specification that excludes continuing severe-distress observations within the recursive forecasting exercise.

## Forecasting Framework

The baseline discriminative specification uses four one-period-lagged prudential predictors:

\[
EAR_{i,t-1},\quad
ROA_{i,t-1},\quad
LLR_{i,t-1},\quad
LIQ_{i,t-1}.
\]

The component models are estimated using an expanding-window forecasting design. For a forecast at time \(t\), the corresponding historical training sample contains observations from years prior to \(t\). The initial country-specific warm-up period is three annual observations.

The persistence-based structural model uses an exponentially weighted moving average of historical distress realizations with smoothing parameter \(\alpha=0.50\), combined with empirical-Bayes-style shrinkage toward the pooled historical distress prevalence. The baseline shrinkage parameter is \(\kappa=2\).

The discriminative component is an \(\ell_2\)-regularized logistic regression with \(C=1\). Predictors are standardized within each recursive training sample prior to ridge-logit estimation.

The HYBRID specification combines the recursively generated STRUCT and LOGIT probabilities on the log-odds scale. Its combination parameters are estimated using leave-one-country-out cross-fitting.

The reported 107 observations constitute the pooled recursive out-of-sample forecast-evaluation panel following the three-year country-specific warm-up period. This figure should not be interpreted as the number of observations with complete values for every continuous prudential indicator. Effective ridge-logit estimation samples vary across recursive windows according to predictor availability.

The persistence-based structural model uses historical distress realizations and therefore does not directly require complete EAR, ROA, LLR, or LIQ observations. For the ridge-logit specification, training observations with incomplete lagged predictors are excluded. When a lagged predictor is unavailable for an individual forecast observation, it is imputed using the median estimated from the corresponding historical training sample.

All stochastic procedures use a fixed random seed (`RANDOM_STATE = 42`) to support reproducibility.

## Calibration and Robustness Procedures

The replication script also reproduces the diagnostic and robustness exercises reported in the paper.

Isotonic probability transformation is applied to the LOGIT and STRUCT predictions in the calibration diagnostics. The HYBRID row reported in the calibration table corresponds to the original cross-fitted HYBRID predictions and is not labelled as isotonic-calibrated.

Forecasting stability is evaluated through repeated bootstrap resampling of the pooled out-of-sample predictions. The procedure reports the mean AUC, the standard deviation of AUC, and the frequency with which each model attains the highest AUC across resamples.

Pairwise differences in AUC are evaluated using nonparametric bootstrap resampling with \(B=1000\) replications. Reported p-values are two-sided bootstrap p-values calculated as twice the smaller proportion of bootstrap AUC differences lying on either side of zero.

Additional robustness analyses include Weight-of-Evidence (WoE) transformation of the lagged prudential predictors, leave-one-country-out influence analysis, the stricter \(S_{it}\geq3\) distress definition, lagged macro-financial proxy augmentation, and sensitivity of the structural model to \(\kappa\in\{1,2,5\}\).

## Repository Contents

### 1. `analysis_dataset.csv`

Analysis-ready country-year panel used for model estimation and forecast evaluation.

The dataset contains country and year identifiers, prudential indicators, constructed banking-distress measures, lagged explanatory variables, and macro-financial covariates used in the forecasting and robustness analyses.

### 2. `EWS_reproducible_Source_Codes.py`

Main Python replication script.

Running the script loads and validates the required structure of the analysis-ready dataset and reproduces the empirical analysis, including:

* recursive component-model estimation and out-of-sample forecasting;
* leave-one-country-out cross-fitted HYBRID probabilities;
* forecasting performance metrics;
* calibration and nonlinear transformation diagnostics;
* bootstrap resampling stability analysis;
* pairwise bootstrap AUC comparisons;
* Weight-of-Evidence robustness analysis;
* coefficient, correlation, and variance-inflation diagnostics;
* leave-one-country-out influence analysis;
* alternative distress definitions;
* macro-financial proxy augmentation; and
* structural shrinkage-parameter sensitivity analysis.

### 3. Software Requirements

The replication code requires Python 3.10 or later and the following Python packages:

* numpy
* pandas
* scipy
* scikit-learn
* statsmodels
* ipython

### 4. `outputs/`

Generated replication results are stored under:

* `outputs/tables/` — manuscript tables and supporting diagnostic results;
* `outputs/predictions/` — out-of-sample prediction files.

## Main Outputs

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

## Reproducibility

To reproduce the analysis, place `analysis_dataset.csv` in the same directory as `EWS_reproducible_Source_Codes.py` and run the script from that directory.

The script automatically creates the required `outputs/tables/` and `outputs/predictions/` directories if they do not already exist.

The replication package begins from the analysis-ready panel. Accordingly, it reproduces the empirical forecasting analysis reported in the paper but does not reconstruct the original manually extracted central-bank source files.
