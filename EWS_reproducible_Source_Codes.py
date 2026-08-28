#!/usr/bin/env python
# coding: utf-8

# # Reproducible recursive OOS EWS experiments
# ## Author: Hedson Malata
# This notebook reproduces the recursive out-of-sample early-warning-system results using the cleaned merged panel dataset only.
# 
# It produces:
# - main recursive OOS performance table for the primary severe-distress target (`y_true`, $(S_{it}$ $\ge$ 2\)),
# - early-fragility environment table (`y_fragility`, reconstructed as $(S_{it} \ge 1))$
# - correlation and VIF diagnostics,
# - recursive logit coefficient summaries,
# - bootstrap AUC difference table for the main target,
# - Monte Carlo forecasting (Recursive forecasting stability under stochastic perturbations) robustness table for the main target,
# - WoE robustness table.
# 

# 
# ## 1. SETTINGS AND REPRODUCIBILITY
# 

# In[1]:


import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from IPython.display import display
from scipy.special import expit
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor


# -------------------------------------------------------------------
# Reproducibility
# -------------------------------------------------------------------

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
random.seed(RANDOM_STATE)
os.environ["PYTHONHASHSEED"] = str(RANDOM_STATE)

# -------------------------------------------------------------------
# Input dataset
# -------------------------------------------------------------------

DATA_FILE = "analysis_dataset.csv"

# -------------------------------------------------------------------
# Output directories
# -------------------------------------------------------------------

OUTPUT_DIR = Path("outputs")
FIG_DIR = OUTPUT_DIR / "figures"
TABLE_DIR = OUTPUT_DIR / "tables"
PRED_DIR = OUTPUT_DIR / "predictions"

for directory in [OUTPUT_DIR, FIG_DIR, TABLE_DIR, PRED_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# -------------------------------------------------------------------
# Recursive forecasting setup
# -------------------------------------------------------------------

WARMUP_YEARS = 3

# -------------------------------------------------------------------
# Structural model hyperparameters
# -------------------------------------------------------------------

STRUCT_ALPHA = 0.50
STRUCT_K = 2.0

# -------------------------------------------------------------------
# Numerical stability
# -------------------------------------------------------------------

EPS = 1e-6

# -------------------------------------------------------------------
# Features used in discriminative model
# -------------------------------------------------------------------

FEATURES = [
    "EAR_lag1",
    "ROA_lag1",
    "LLR_lag1",
    "LIQ_lag1",
]

# -------------------------------------------------------------------
# Distress targets
# -------------------------------------------------------------------

TARGET_MAIN = "y_true"
TARGET_EARLY_FRAGILITY = "y_fragility"

print("Settings loaded successfully.")
print(f"Dataset: {DATA_FILE}")
print(f"Main target: {TARGET_MAIN}")
print(f"Early-fragility target: {TARGET_EARLY_FRAGILITY}")
print(f"Features: {FEATURES}")


# In[2]:


# =========================================================
# 2. GENERAL HELPER FUNCTIONS
# =========================================================

def safe_prob(p, eps=EPS):
    """Clip probabilities away from exactly 0 and 1."""
    return np.clip(np.asarray(p, dtype=float), eps, 1.0 - eps)


def safe_logit(p, eps=EPS):
    """Numerically stable logit transform."""
    p = safe_prob(p, eps=eps)
    return np.log(p) - np.log1p(-p)


def sigmoid(z):
    """Inverse-logit transform."""
    return expit(z)


def make_logit_pipeline():
    """Ridge-logit pipeline used in the recursive discriminative model."""
    return make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        LogisticRegression(
            penalty="l2",
            solver="lbfgs",
            max_iter=1000,
            random_state=RANDOM_STATE,
        ),
    )


def make_oos_dates(df, warmup=WARMUP_YEARS):
    """Return country-year pairs used for recursive out-of-sample evaluation."""
    rows = []

    for country, group in df.groupby("country"):
        years = np.sort(group["year"].dropna().unique())
        if len(years) > warmup:
            rows.append(pd.DataFrame({"country": country, "year": years[warmup:]}))

    if not rows:
        return pd.DataFrame(columns=["country", "year"])

    return pd.concat(rows, ignore_index=True)


def evaluate_prediction_table(df, target_col, model_map):
    """Compute Brier, LogLoss, RMSE, AUC, AP, N, events and prevalence."""
    rows = []

    for model_name, p_col in model_map.items():
        if p_col not in df.columns:
            continue

        tmp = df[[target_col, p_col]].dropna().copy()
        if tmp.empty:
            continue

        y = tmp[target_col].astype(int).values
        p = safe_prob(tmp[p_col].astype(float).values)
        brier = brier_score_loss(y, p)

        if len(np.unique(y)) < 2:
            auc_val = np.nan
            ap_val = np.nan
        else:
            auc_val = roc_auc_score(y, p)
            ap_val = average_precision_score(y, p)

        rows.append({
            "Model": model_name,
            "Brier": brier,
            "LogLoss": log_loss(y, p, labels=[0, 1]),
            "RMSE": np.sqrt(brier),
            "AUC": auc_val,
            "AP": ap_val,
            "N": len(y),
            "Events": int(y.sum()),
            "Prevalence": y.mean(),
        })

    out = pd.DataFrame(rows)
    if out.empty:
        return out

    return out.sort_values("AUC", ascending=False).reset_index(drop=True)


def rounded_table(df, digits=3):
    """Round numeric reporting columns without changing saved raw results elsewhere."""
    out = df.copy()
    cols = ["Brier", "LogLoss", "RMSE", "AUC", "AP", "Prevalence"]
    cols = [c for c in cols if c in out.columns]
    out[cols] = out[cols].round(digits)
    return out


# In[3]:


# =========================================================
# 3. LOAD AND VALIDATE DATASET
# =========================================================

def load_panel(data_file=DATA_FILE):
    """Load the merged panel dataset and validate required columns.

    The paper treats early fragility as a separate distress environment,
    It is reconstructed directly as
    y_fragility = 1{stress_score >= 1}.
    """
    panel = pd.read_csv(data_file)

    required_cols = [
        "country", "year",
        "EAR", "ROA", "LLR", "LIQ", "WFS",
        "frag_capital", "frag_profit", "frag_assetq", "frag_liquidity",
        "stress_score", TARGET_MAIN,
        *FEATURES,
    ]

    missing_cols = [col for col in required_cols if col not in panel.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns in {data_file}: {missing_cols}")

    panel["year"] = pd.to_numeric(panel["year"], errors="coerce")
    if panel["year"].isna().any():
        raise ValueError("Column 'year' contains non-numeric or missing values.")

    panel["year"] = panel["year"].astype(int)

    # Early fragility environment used in the paper: S_it >= 1.
    panel[TARGET_EARLY_FRAGILITY] = (
        pd.to_numeric(panel["stress_score"], errors="coerce") >= 1
    ).astype(int)

    panel = panel.sort_values(["country", "year"]).reset_index(drop=True)

    return panel


panel = load_panel(DATA_FILE)
oos_dates = make_oos_dates(panel, warmup=WARMUP_YEARS)

if oos_dates.empty:
    raise RuntimeError("No country has enough years for recursive out-of-sample evaluation.")

print("Loaded dataset:", DATA_FILE)
print("Panel shape:", panel.shape)
print("Countries:", sorted(panel["country"].unique()))
print("Years:", int(panel["year"].min()), "-", int(panel["year"].max()))
print("OOS rows:", len(oos_dates))

print("\nSevere-distress target distribution:")
print(panel[TARGET_MAIN].value_counts(normalize=True).sort_index())

print("\nEarly-fragility target distribution:")
print(panel[TARGET_EARLY_FRAGILITY].value_counts(normalize=True).sort_index())


# In[4]:


# =========================================================
# 4. RECURSIVE FORECASTING MODELS
# =========================================================

def historical_prior(df, year, target_col):
    """Historical pooled event rate before the forecast year."""
    hist = df[df["year"] < year]
    prior = hist[target_col].mean()

    if pd.isna(prior):
        prior = df[target_col].mean()
    if pd.isna(prior):
        prior = 0.5

    return float(safe_prob(prior))


def compute_naive_prob(df, country, year, target_col):
    """Naive pooled historical prior."""
    return historical_prior(df, year, target_col)


def compute_lag_prob(df, country, year, target_col):
    """One-period country-specific lag benchmark, falling back to pooled history."""
    row = df[(df["country"] == country) & (df["year"] == year)]
    if row.empty:
        return np.nan

    country_hist = df[(df["country"] == country) & (df["year"] < year)].sort_values("year")

    if not country_hist.empty:
        lag_value = country_hist[target_col].iloc[-1]
        if pd.notna(lag_value):
            return float(safe_prob(lag_value))

    return historical_prior(df, year, target_col)


def compute_structural_prob(df, country, year, target_col, alpha=STRUCT_ALPHA, k=STRUCT_K):
    """Persistence-based structural probability with empirical-Bayes shrinkage."""
    hist_all = df[df["year"] < year].copy()
    if hist_all.empty:
        return 0.5

    global_prior = hist_all[target_col].mean()
    if pd.isna(global_prior):
        global_prior = 0.5

    hist_country = hist_all[hist_all["country"] == country].sort_values("year").copy()
    if hist_country.empty:
        return float(safe_prob(global_prior))

    ewma = hist_country[target_col].ewm(alpha=alpha, adjust=False).mean()
    n_hist = len(hist_country)
    shrinkage_weight = n_hist / (n_hist + k)

    m_last = ewma.iloc[-1]
    if pd.isna(m_last):
        m_last = global_prior

    p_struct = shrinkage_weight * m_last + (1.0 - shrinkage_weight) * global_prior
    return float(safe_prob(p_struct))


def recursive_logit_prob(df, country, year, features, target_col):
    """Recursive pooled ridge-logit probability."""
    train = df[df["year"] < year].copy()
    test = df[(df["country"] == country) & (df["year"] == year)].copy()

    if test.empty:
        return np.nan

    train_model = train.dropna(subset=features + [target_col]).copy()
    y_train = train_model[target_col].astype(int)

    if len(train_model) < 8 or y_train.nunique() < 2:
        return historical_prior(df, year, target_col)

    pipe = make_logit_pipeline()
    pipe.fit(train_model[features], y_train)

    return float(safe_prob(pipe.predict_proba(test[features])[0, 1]))


def build_recursive_predictions(df, oos_dates, features, target_col):
    """Build OOS probabilities for structural, logit, naive and lag models."""
    rows = []

    for _, row in oos_dates.iterrows():
        country = row["country"]
        year = int(row["year"])

        test = df[(df["country"] == country) & (df["year"] == year)]
        if test.empty:
            continue

        rows.append({
            "country": country,
            "year": year,
            "h": 1,
            target_col: int(test[target_col].iloc[0]),
            "p_struct": compute_structural_prob(df, country, year, target_col),
            "p_logit": recursive_logit_prob(df, country, year, features, target_col),
            "p_naive": compute_naive_prob(df, country, year, target_col),
            "p_lag": compute_lag_prob(df, country, year, target_col),
        })

    return pd.DataFrame(rows).sort_values(["country", "year"]).reset_index(drop=True)


# In[5]:


# =========================================================
# 5. CROSS-FITTED HYBRID MODEL
# =========================================================

def fit_offset_glm(lp_struct, lp_logit, y, alpha=1e-4):
    """
    Estimate log-odds pooling:
        logit(p_hybrid) = logit(p_struct) + a + w[logit(p_logit)-logit(p_struct)]
    with w clipped to [0, 1].
    """
    X = pd.DataFrame({"delta": lp_logit - lp_struct})
    X = sm.add_constant(X, has_constant="add")

    glm = sm.GLM(
        y.astype(int),
        X,
        family=sm.families.Binomial(),
        offset=lp_struct,
    )

    try:
        res = glm.fit_regularized(alpha=alpha, L1_wt=0.0)
    except Exception:
        res = glm.fit()

    a = float(res.params["const"])
    w = float(np.clip(res.params["delta"], 0.0, 1.0))

    return w, a


def cross_fitted_hybrid(oos_df, target_col):
    """Leave-one-country-out stacking for hybrid OOS probabilities."""
    df = oos_df.dropna(subset=[target_col, "p_struct", "p_logit", "country"]).copy()

    if df.empty:
        return {"mode": "empty", "a": np.nan, "w": np.nan}, df

    df["p_struct"] = safe_prob(df["p_struct"].values)
    df["p_logit"] = safe_prob(df["p_logit"].values)

    if len(df) < 8 or df[target_col].nunique() < 2:
        df["p_hybrid_cv"] = 0.5 * df["p_struct"] + 0.5 * df["p_logit"]
        return {"mode": "average_fallback", "a": 0.0, "w": 0.5}, df

    y = df[target_col].astype(int).values
    lp_struct = safe_logit(df["p_struct"].values)
    lp_logit = safe_logit(df["p_logit"].values)

    df["p_hybrid_cv"] = np.nan

    for country in df["country"].unique():
        test_mask = df["country"].values == country
        train_mask = ~test_mask

        if train_mask.sum() < 6 or np.unique(y[train_mask]).size < 2:
            df.loc[test_mask, "p_hybrid_cv"] = (
                0.5 * df.loc[test_mask, "p_struct"].values
                + 0.5 * df.loc[test_mask, "p_logit"].values
            )
            continue

        w, a = fit_offset_glm(lp_struct[train_mask], lp_logit[train_mask], y[train_mask])
        z = a + w * (lp_logit[test_mask] - lp_struct[test_mask]) + lp_struct[test_mask]
        df.loc[test_mask, "p_hybrid_cv"] = safe_prob(sigmoid(z))

    w_all, a_all = fit_offset_glm(lp_struct, lp_logit, y)
    return {"mode": "leave_one_country_out_offset_glm", "a": a_all, "w": w_all}, df


# In[6]:


# =========================================================
# 6. COEFFICIENTS AND BOOTSTRAP HELPERS
# =========================================================

def expanding_logit_coefficients(df, oos_dates, features, target_col):
    """Store standardized recursive ridge-logit coefficients for each forecast year."""
    coef_rows = []

    for year in sorted(oos_dates["year"].unique()):
        train = df[df["year"] < int(year)].copy()
        train_model = train.dropna(subset=features + [target_col]).copy()

        if len(train_model) < 8:
            continue

        y_train = train_model[target_col].astype(int)
        if y_train.nunique() < 2:
            continue

        pipe = make_logit_pipeline()
        pipe.fit(train_model[features], y_train)
        coefs = pipe.named_steps["logisticregression"].coef_[0]

        coef_rows.append({"forecast_year": int(year), **dict(zip(features, coefs))})

    return pd.DataFrame(coef_rows)


def summarize_coefficients(coef_df, features):
    if coef_df.empty:
        return pd.DataFrame(columns=["Variable", "Mean", "Std. Dev.", "Sign"])

    out = (
        coef_df[features]
        .agg(["mean", "std"])
        .T
        .reset_index()
        .rename(columns={"index": "Variable", "mean": "Mean", "std": "Std. Dev."})
    )
    out["Sign"] = np.where(out["Mean"] >= 0, "+", "-")
    return out


def bootstrap_auc_diff(y, p1, p2, n_boot=1000, random_state=RANDOM_STATE):
    """Bootstrap difference in AUC: AUC(p1) - AUC(p2)."""
    rng = np.random.default_rng(random_state)
    y = np.asarray(y).astype(int)
    p1 = np.asarray(p1).astype(float)
    p2 = np.asarray(p2).astype(float)

    diffs = []
    n = len(y)

    for _ in range(n_boot):
        idx = rng.choice(n, size=n, replace=True)
        y_b = y[idx]

        if len(np.unique(y_b)) < 2:
            continue

        diffs.append(roc_auc_score(y_b, p1[idx]) - roc_auc_score(y_b, p2[idx]))

    diffs = np.asarray(diffs)
    if len(diffs) == 0:
        return {
            "mean_diff": np.nan,
            "ci_lower": np.nan,
            "ci_upper": np.nan,
            "p_value": np.nan,
            "n_boot_used": 0,
        }

    p_value = 2 * min(np.mean(diffs <= 0), np.mean(diffs >= 0))

    return {
        "mean_diff": float(np.mean(diffs)),
        "ci_lower": float(np.percentile(diffs, 2.5)),
        "ci_upper": float(np.percentile(diffs, 97.5)),
        "p_value": min(float(p_value), 1.0),
        "n_boot_used": int(len(diffs)),
    }


def bootstrap_auc_table(oos_preds, target_col):
    comparisons = [
        ("p_struct", "p_logit", "STRUCT vs LOGIT"),
        ("p_struct", "p_hybrid_cv", "STRUCT vs HYBRID"),
        ("p_logit", "p_hybrid_cv", "LOGIT vs HYBRID"),
        ("p_naive", "p_lag", "NAIVE vs LAG"),
    ]

    rows = []

    for c1, c2, label in comparisons:
        mask = oos_preds[[target_col, c1, c2]].notna().all(axis=1)

        if mask.sum() < 3 or oos_preds.loc[mask, target_col].nunique() < 2:
            continue

        res = bootstrap_auc_diff(
            y=oos_preds.loc[mask, target_col].astype(int).values,
            p1=oos_preds.loc[mask, c1].astype(float).values,
            p2=oos_preds.loc[mask, c2].astype(float).values,
            n_boot=1000,
            random_state=RANDOM_STATE,
        )

        rows.append({
            "Comparison": label,
            "Mean Diff.": res["mean_diff"],
            "Lower CI": res["ci_lower"],
            "Upper CI": res["ci_upper"],
            "p-value": res["p_value"],
            "n_boot_used": res["n_boot_used"],
        })

    return pd.DataFrame(rows)


# In[7]:


# =========================================================
# 7. SINGLE EXPERIMENT RUNNER
# =========================================================

MODEL_MAP = {
    "STRUCT": "p_struct",
    "HYBRID": "p_hybrid_cv",
    "LOGIT": "p_logit",
    "NAIVE": "p_naive",
    "LAG": "p_lag",
}


def run_experiment(panel, oos_dates, target_col, label):
    """Run the full recursive OOS experiment for one target definition."""
    print(f"\nRunning recursive OOS experiment for {label}: {target_col}")

    oos_base = build_recursive_predictions(
        df=panel,
        oos_dates=oos_dates,
        features=FEATURES,
        target_col=target_col,
    )

    fit_info, oos_preds = cross_fitted_hybrid(oos_base, target_col=target_col)

    table = evaluate_prediction_table(
        df=oos_preds,
        target_col=target_col,
        model_map=MODEL_MAP,
    )


    # Save raw predictions and rounded reporting tables.
    oos_preds.to_csv(PRED_DIR / f"oos_predictions_{label}.csv", index=False)
    rounded_table(table).to_csv(TABLE_DIR / f"table_recursive_oos_{label}.csv", index=False)
    
    print("Hybrid fit:", fit_info)
    print(f"Saved predictions: {PRED_DIR / f'oos_predictions_{label}.csv'}")
    print(f"Saved table: {TABLE_DIR / f'table_recursive_oos_{label}.csv'}")

    return {
        "fit_info": fit_info,
        "oos_predictions": oos_preds,
        "table": table,
    }


# In[8]:


# =========================================================
# 8. Recursive experiment runner
# =========================================================

MODEL_MAP = {
    "STRUCT": "p_struct",
    "HYBRID": "p_hybrid_cv",
    "LOGIT": "p_logit",
    "NAIVE": "p_naive",
    "LAG": "p_lag",
}

# ---------------------------------------------------------
# Hybrid calibration settings
# ---------------------------------------------------------

USE_ISOTONIC = True
ISO_BLEND = 0.10

def run_recursive_experiment(
    panel,
    target_col,
    features,
):
    """
    Run recursive OOS experiment for a given target definition.
    """

    # -----------------------------------------------------
    # Build OOS dates
    # -----------------------------------------------------

    oos_dates_local = make_oos_dates(
        panel,
        warmup=WARMUP_YEARS,
    )

    # -----------------------------------------------------
    # Structural model
    # -----------------------------------------------------

    oos_struct_local = structural_oos(
        panel,
        oos_dates_local,
        target_col=target_col,
        alpha=STRUCT_ALPHA,
        k=STRUCT_K,
    )

    # -----------------------------------------------------
    # Logistic model
    # -----------------------------------------------------

    oos_logit_local = logit_oos_recursive_pooled(
        panel,
        oos_dates_local,
        features,
        target_col=target_col,
    )

    # -----------------------------------------------------
    # Merge predictions
    # -----------------------------------------------------

    oos_local = (
        oos_struct_local
        .merge(
            oos_logit_local,
            on=["country", "year", "h"],
            how="left",
        )
        .sort_values(["country", "year"])
        .reset_index(drop=True)
    )

    # -----------------------------------------------------
    # Lag target
    # -----------------------------------------------------

    panel_local = panel.copy()

    panel_local["prev_target"] = (
        panel_local
        .groupby("country")[target_col]
        .shift(1)
    )

    # -----------------------------------------------------
    # Benchmarks
    # -----------------------------------------------------

    oos_local["p_naive"] = [
        compute_naive_prob(
            panel_local,
            c,
            y,
            target_col=target_col,
        )
        for c, y in zip(
            oos_local["country"],
            oos_local["year"],
        )
    ]

    oos_local["p_lag"] = [
        compute_lag_prob(
            panel_local,
            c,
            y,
            target_col=target_col,
        )
        for c, y in zip(
            oos_local["country"],
            oos_local["year"],
        )
    ]

    # -----------------------------------------------------
    # Hybrid
    # -----------------------------------------------------
    fit_info, oos_preds_local = cross_fitted_hybrid(
        oos_local,
        target_col=target_col,
    )
    # -----------------------------------------------------
    # Evaluation table
    # -----------------------------------------------------

    results_table = evaluate_prediction_table(
        df=oos_preds_local,
        target_col=target_col,
        model_map=MODEL_MAP,
    )

    return (
        oos_local,
        results_table,
        oos_preds_local,
    )


# ## Main tables used in the paper

# In[9]:


def logit_oos_recursive_pooled(
    df,
    oos_dates,
    features,
    target_col="y_true",
):
    rows = []

    for _, row in oos_dates.iterrows():
        c = row["country"]
        y_star = int(row["year"])

        tr = df[df["year"] < y_star].copy()
        te = df[
            (df["country"] == c)
            & (df["year"] == y_star)
        ].copy()

        if te.empty:
            continue

        tr_model = tr.dropna(subset=features + [target_col]).copy()
        ytr = tr_model[target_col].astype(int)

        if len(tr_model) < 8 or ytr.nunique() < 2:
            prior = tr[target_col].mean()

            if pd.isna(prior):
                prior = df[target_col].mean()

            if pd.isna(prior):
                prior = 0.5

            p = prior

        else:
            pipe = make_logit_pipeline()
            pipe.fit(tr_model[features], ytr)
            p = pipe.predict_proba(te[features])[0, 1]

        rows.append({
            "country": c,
            "year": y_star,
            "h": 1,
            "p_logit": float(np.clip(p, EPS, 1.0 - EPS)),
        })

    return pd.DataFrame(rows)
# =========================================================
# South Africa exclusion robustness
# =========================================================

def structural_oos(
    df,
    oos_dates,
    target_col="y_true",
    alpha=0.5,
    k=2.0,
):
    rows = []

    for _, row in oos_dates.iterrows():
        c = row["country"]
        y_star = int(row["year"])

        test_row = df[
            (df["country"] == c)
            & (df["year"] == y_star)
        ]

        if test_row.empty:
            continue

        p_struct = compute_structural_prob(
            df,
            c,
            y_star,
            target_col=target_col,
            alpha=alpha,
            k=k,
        )

        rows.append({
            "country": c,
            "year": y_star,
            "h": 1,
            target_col: int(test_row[target_col].iloc[0]),
            "p_struct": float(np.clip(p_struct, EPS, 1.0 - EPS)),
        })

    return pd.DataFrame(rows)


# ## Table 1: Out-of-sample model performance

# In[10]:


# =========================================================
# MAIN RECURSIVE EXPERIMENT
# =========================================================
oos_main, results_main, _ = run_recursive_experiment(
    panel=panel,
    target_col=TARGET_MAIN,
    features=FEATURES,
)

main_results = {
    "oos_predictions": oos_main,
    "model_performance": results_main,
}

print("\nTable 1: MAIN MODEL PERFORMANCE")
display(results_main)


# ## TABLE 2: Forecasting performance across distress environments

# In[11]:


# ---------------------------------------------------------
# Construct purified distress target
# Purified distress = severe distress onset only
# Continuing severe-distress observations are removed
# ---------------------------------------------------------

panel = panel.sort_values(["country", "year"]).copy()

panel["y_true_lag"] = panel.groupby("country")["y_true"].shift(1)

panel["continuing_severe"] = (
    (panel["y_true"] == 1) &
    (panel["y_true_lag"] == 1)
)

panel_purified = panel.loc[~panel["continuing_severe"]].copy()

TARGET_PURIFIED = "y_true"

# ---------------------------------------------------------
# Run recursive experiments
# ---------------------------------------------------------

environment_specs = [
    (
        "Severe distress\n($S_{it} \\geq 2$)",
        panel,
        TARGET_MAIN,
    ),
    (
        "Early fragility\n($S_{it} \\geq 1$)",
        panel,
        TARGET_EARLY_FRAGILITY,
    ),
    (
        "Purified distress",
        panel_purified,
        TARGET_PURIFIED,
    ),
]

environment_rows = []

for env_name, env_panel, target_col in environment_specs:

    oos_env, results_env, _ = run_recursive_experiment(
        panel=env_panel,
        target_col=target_col,
        features=FEATURES,
    )

    results_env = results_env[
        results_env["Model"].isin(["STRUCT", "HYBRID", "LOGIT"])
    ].copy()

    results_env.insert(0, "Environment", env_name)

    environment_rows.append(
        results_env[["Environment", "Model", "AUC", "AP", "Brier"]]
    )

# ---------------------------------------------------------
# Combine results
# ---------------------------------------------------------

table_2 = pd.concat(
    environment_rows,
    axis=0,
    ignore_index=True,
)

# ---------------------------------------------------------
# Order rows
# ---------------------------------------------------------

env_order = {
    "Severe distress\n($S_{it} \\geq 2$)": 1,
    "Early fragility\n($S_{it} \\geq 1$)": 2,
    "Purified distress": 3,
}

model_order = {
    "STRUCT": 1,
    "HYBRID": 2,
    "LOGIT": 3,
}

table_2["env_order"] = table_2["Environment"].map(env_order)
table_2["model_order"] = table_2["Model"].map(model_order)

table_2 = (
    table_2
    .sort_values(["env_order", "model_order"])
    .drop(columns=["env_order", "model_order"])
    .reset_index(drop=True)
)

# ---------------------------------------------------------
# Round and display
# ---------------------------------------------------------

table_2[["AUC", "AP", "Brier"]] = table_2[["AUC", "AP", "Brier"]].round(3)

print("\nTABLE 2: Forecasting performance across distress environments")
display(table_2)

# ---------------------------------------------------------
# Save
# ---------------------------------------------------------

table_2.to_csv(
    os.path.join(TABLE_DIR, "table_2_distress_environments.csv"),
    index=False
)

print("Table 2 saved successfully.")


# ### TABLE 3: Calibration and nonlinear transformation diagnostics

# In[12]:


# ---------------------------------------------------------
# Baseline severe-distress predictions
# ---------------------------------------------------------

oos_base_t3, table_base_t3, oos_preds_t3 = run_recursive_experiment(
    panel=panel,
    target_col=TARGET_MAIN,
    features=FEATURES,
)

# ---------------------------------------------------------
# Isotonic calibration
# ---------------------------------------------------------

from sklearn.isotonic import IsotonicRegression

y_true_t3 = oos_preds_t3[TARGET_MAIN].values

# ---------------------------------------------------------
# LOGIT isotonic
# ---------------------------------------------------------

iso_logit = IsotonicRegression(
    out_of_bounds="clip"
)

p_logit_iso = iso_logit.fit_transform(
    oos_preds_t3["p_logit"],
    y_true_t3,
)

# ---------------------------------------------------------
# HYBRID isotonic
# ---------------------------------------------------------

iso_hybrid = IsotonicRegression(
    out_of_bounds="clip"
)

p_hybrid_iso = oos_preds_t3["p_hybrid_cv"].values
# ---------------------------------------------------------
# STRUCT isotonic
# ---------------------------------------------------------

iso_struct = IsotonicRegression(
    out_of_bounds="clip"
)

p_struct_iso = iso_struct.fit_transform(
    oos_preds_t3["p_struct"],
    y_true_t3,
)

# ---------------------------------------------------------
# Build calibrated prediction frame
# ---------------------------------------------------------

table3_df = pd.DataFrame({
    TARGET_MAIN: y_true_t3,

    "LOGIT": oos_preds_t3["p_logit"],
    "LOGIT_ISO": p_logit_iso,

    "HYBRID_ISO": p_hybrid_iso,
    "STRUCT_ISO": p_struct_iso,
})

# ---------------------------------------------------------
# Evaluation helper
# ---------------------------------------------------------

from sklearn.metrics import (
    roc_auc_score,
    brier_score_loss,
)

def calibration_row(y, p, name):

    return {
        "Model": name,
        "AUC": roc_auc_score(y, p),
        "Brier": brier_score_loss(y, p),
        "Mean Pred.": np.mean(p),
    }

# ---------------------------------------------------------
# Generate rows
# ---------------------------------------------------------

table3_rows = []

table3_rows.append(
    calibration_row(
        y_true_t3,
        table3_df["LOGIT"],
        "LOGIT",
    )
)

table3_rows.append(
    calibration_row(
        y_true_t3,
        table3_df["LOGIT_ISO"],
        "LOGIT_ISO",
    )
)

table3_rows.append(
    calibration_row(
        y_true_t3,
        table3_df["HYBRID_ISO"],
        "HYBRID_ISO",
    )
)

table3_rows.append(
    calibration_row(
        y_true_t3,
        table3_df["STRUCT_ISO"],
        "STRUCT_ISO",
    )
)

table_3 = pd.DataFrame(table3_rows)

# ---------------------------------------------------------
# Round for manuscript display
# ---------------------------------------------------------

table_3[["AUC", "Brier", "Mean Pred."]] = (
    table_3[["AUC", "Brier", "Mean Pred."]]
    .round(3)
)

print("\nTABLE 3: Calibration and nonlinear transformation diagnostics")
display(table_3)

# ---------------------------------------------------------
# Save
# ---------------------------------------------------------

table_3.to_csv(
    os.path.join(
        TABLE_DIR,
        "table_3_calibration_diagnostics.csv",
    ),
    index=False,
)

print("Table 3 saved successfully.")


# 
# ### TABLE 4: TABLE 4: Recursive forecasting stability under stochastic perturbation
# 

# In[13]:


oos_main, results_main, oos_preds_main = run_recursive_experiment(
    panel=panel,
    target_col=TARGET_MAIN,
    features=FEATURES,
)

main_results = {
    "oos_raw": oos_main,
    "results_table": results_main,
    "oos_predictions": oos_preds_main,
}

def resampling_forecasting_stability(
    oos_preds,
    target_col="y_true",
    model_map=None,
    n_iter=1000,
    random_state=RANDOM_STATE,
):

    if model_map is None:
        model_map = {
            "STRUCT": "p_struct",
            "HYBRID": "p_hybrid_cv",
            "LOGIT": "p_logit",
        }

    required = [target_col, *model_map.values()]

    missing_cols = [col for col in required if col not in oos_preds.columns]
    if missing_cols:
        raise KeyError(
            f"Missing required columns: {missing_cols}. "
            f"Available columns are: {list(oos_preds.columns)}"
        )

    df = oos_preds[required].dropna().copy()
    y = df[target_col].astype(int).to_numpy()

    p = {
        model: safe_prob(df[col].astype(float).to_numpy())
        for model, col in model_map.items()
    }

    print("\nDirect AUC check:")
    for model, probs in p.items():
        print(
            model, ":",
            round(roc_auc_score(y, probs), 3),
            "| min:", round(probs.min(), 4),
            "| max:", round(probs.max(), 4),
        )

    rng = np.random.default_rng(random_state)
    rows = []
    n = len(df)

    for b in range(n_iter):
        idx = rng.choice(n, size=n, replace=True)
        y_b = y[idx]

        if np.unique(y_b).size < 2:
            continue

        aucs = {
            model: roc_auc_score(y_b, probs[idx])
            for model, probs in p.items()
        }

        winner = max(aucs, key=aucs.get)

        for model, auc_value in aucs.items():
            rows.append({
                "iteration": b + 1,
                "Model": model,
                "AUC": auc_value,
                "Winner": model == winner,
            })

    draws = pd.DataFrame(rows)

    summary = (
        draws.groupby("Model")
        .agg(
            **{
                "Mean AUC": ("AUC", "mean"),
                "SD(AUC)": ("AUC", "std"),
                "Win Freq.": ("Winner", "mean"),
            }
        )
        .reset_index()
    )

    order = ["HYBRID", "LOGIT", "STRUCT"]
    summary["Model"] = pd.Categorical(
        summary["Model"],
        categories=order,
        ordered=True,
    )

    summary = summary.sort_values("Model").reset_index(drop=True)
    summary["Model"] = summary["Model"].astype(str)

    return summary, draws


mc_table, mc_draws = resampling_forecasting_stability(
    oos_preds=main_results["oos_predictions"],
    target_col="y_true",
    model_map={
        "STRUCT": "p_struct",
        "HYBRID": "p_hybrid_cv",
        "LOGIT": "p_logit",
    },
    n_iter=1000,
    random_state=RANDOM_STATE,
)

mc_table_display = mc_table.copy()
mc_table_display["Mean AUC"] = mc_table_display["Mean AUC"].round(3)
mc_table_display["SD(AUC)"] = mc_table_display["SD(AUC)"].round(3)
mc_table_display["Win Freq."] = (
    100 * mc_table_display["Win Freq."]
).round(1).astype(str) + "%"

print("\nTABLE 4: Recursive forecasting stability under stochastic perturbation")
display(mc_table_display)


# ### TABLE 5: Robustness analysis using WoE-transformed predictors

# In[14]:


def make_quantile_bins(x, n_bins=4):
    return pd.qcut(x, q=n_bins, duplicates="drop")


def compute_woe_map(x_binned, y, eps=0.5):
    tmp = pd.DataFrame({"bin": x_binned, "y": y}).dropna()

    total_good = (tmp["y"] == 0).sum()
    total_bad = (tmp["y"] == 1).sum()

    woe_map = {}
    for bin_label, group in tmp.groupby("bin", observed=False):
        good = (group["y"] == 0).sum()
        bad = (group["y"] == 1).sum()

        good_share = (good + eps) / (total_good + eps)
        bad_share = (bad + eps) / (total_bad + eps)
        woe_map[bin_label] = np.log(good_share / bad_share)

    return woe_map


def apply_woe(train_x, train_y, test_x, n_bins=4):
    train_bins = make_quantile_bins(train_x, n_bins=n_bins)
    categories = train_bins.cat.categories

    if len(categories) < 2:
        return (
            pd.Series(np.zeros(len(train_x)), index=train_x.index),
            pd.Series(np.zeros(len(test_x)), index=test_x.index),
        )

    edges = [categories[0].left] + [cat.right for cat in categories]
    edges[0] = -np.inf
    edges[-1] = np.inf

    train_bins = pd.cut(train_x, bins=edges, include_lowest=True)
    test_bins = pd.cut(test_x, bins=edges, include_lowest=True)

    woe_map = compute_woe_map(train_bins, train_y)

    train_woe = train_bins.map(woe_map).astype(float).fillna(0.0)
    test_woe = test_bins.map(woe_map).astype(float).fillna(0.0)

    return train_woe, test_woe


def recursive_woe_logit_oos(df, oos_dates, features, target_col, n_bins=4):
    rows = []
    coef_rows = []

    for _, row in oos_dates.iterrows():
        country = row["country"]
        year = int(row["year"])

        train = df[df["year"] < year].copy()
        test = df[(df["country"] == country) & (df["year"] == year)].copy()

        if test.empty:
            continue

        train_model = train.dropna(subset=features + [target_col]).copy()
        y_train = train_model[target_col].astype(int)

        if len(train_model) < 8 or y_train.nunique() < 2:
            p = historical_prior(df, year, target_col)
            rows.append({
                "country": country,
                "year": year,
                "h": 1,
                target_col: int(test[target_col].iloc[0]),
                "p_logit_woe": p,
            })
            continue

        X_train_woe = pd.DataFrame(index=train_model.index)
        X_test_woe = pd.DataFrame(index=test.index)

        for feature in features:
            train_woe, test_woe = apply_woe(
                train_model[feature],
                y_train,
                test[feature],
                n_bins=n_bins,
            )
            X_train_woe[f"{feature}_woe"] = train_woe
            X_test_woe[f"{feature}_woe"] = test_woe

        model = make_logit_pipeline()
        model.fit(X_train_woe, y_train)
        p = model.predict_proba(X_test_woe)[0, 1]

        rows.append({
            "country": country,
            "year": year,
            "h": 1,
            target_col: int(test[target_col].iloc[0]),
            "p_logit_woe": float(safe_prob(p)),
        })

        coefs = model.named_steps["logisticregression"].coef_[0]
        coef_rows.append({
            "forecast_year": year,
            **dict(zip(X_train_woe.columns, coefs)),
        })

    return pd.DataFrame(rows), pd.DataFrame(coef_rows)


oos_woe, coef_woe = recursive_woe_logit_oos(
    df=panel,
    oos_dates=oos_dates,
    features=FEATURES,
    target_col=TARGET_MAIN,
    n_bins=4,
)

oos_compare = main_results["oos_predictions"].merge(
    oos_woe[["country", "year", "p_logit_woe"]],
    on=["country", "year"],
    how="left",
)

woe_model_map = {
    "STRUCT": "p_struct",
    "HYBRID": "p_hybrid_cv",
    "Raw logit": "p_logit",
    "WoE logit": "p_logit_woe",
}

table_woe = evaluate_prediction_table(
    df=oos_compare,
    target_col=TARGET_MAIN,
    model_map=woe_model_map,
)

table_woe_display = rounded_table(table_woe)
table_woe_display.to_csv(TABLE_DIR / "table_woe_robustness_main.csv", index=False)
oos_compare.to_csv(PRED_DIR / "oos_predictions_main_with_woe.csv", index=False)
coef_woe.to_csv(TABLE_DIR / "woe_logit_coefficients_main.csv", index=False)

print("Table 5: WoE robustness table: main target")
display(table_woe_display)


# ### TABLE 6. BOOTSTRAP AUC DIFFERENCES

# In[15]:


def bootstrap_auc_table(
    oos_preds,
    target_col="y_true",
    model_map=None,
    n_boot=1000,
    random_state=RANDOM_STATE,
):
    """
    Nonparametric bootstrap AUC differences using finalized recursive
    out-of-sample predictions.
    """

    if model_map is None:
        model_map = {
            "STRUCT": "p_struct",
            "LOGIT": "p_logit",
            "HYBRID": "p_hybrid_cv",
            "NAIVE": "p_naive",
            "LAG": "p_lag",
        }

    required = [target_col, *model_map.values()]

    missing_cols = [col for col in required if col not in oos_preds.columns]
    if missing_cols:
        raise KeyError(
            f"Missing required columns: {missing_cols}. "
            f"Available columns are: {list(oos_preds.columns)}"
        )

    df = oos_preds[required].dropna().copy()

    if df.empty:
        raise ValueError("No complete OOS predictions available.")

    y = df[target_col].astype(int).to_numpy()

    if np.unique(y).size < 2:
        raise ValueError(
            "Target has only one class. AUC cannot be computed."
        )

    probs = {
        model: safe_prob(df[col].astype(float).to_numpy())
        for model, col in model_map.items()
    }

    comparisons = [
        ("STRUCT", "LOGIT", "STRUCT-LOGIT"),
        ("STRUCT", "HYBRID", "STRUCT-HYBRID"),
        ("LOGIT", "HYBRID", "LOGIT-HYBRID"),
        ("NAIVE", "LAG", "NAIVE-LAG"),
    ]

    rng = np.random.default_rng(random_state)
    n = len(df)

    rows = []

    for model_1, model_2, label in comparisons:

        observed_diff = (
            roc_auc_score(y, probs[model_1])
            - roc_auc_score(y, probs[model_2])
        )

        boot_diffs = []

        for _ in range(n_boot):

            idx = rng.choice(
                n,
                size=n,
                replace=True,
            )

            y_b = y[idx]

            if np.unique(y_b).size < 2:
                continue

            auc_1 = roc_auc_score(
                y_b,
                probs[model_1][idx],
            )

            auc_2 = roc_auc_score(
                y_b,
                probs[model_2][idx],
            )

            boot_diffs.append(auc_1 - auc_2)

        boot_diffs = np.asarray(boot_diffs)

        if boot_diffs.size == 0:
            continue

        rows.append(
            {
                "Comparison": label,
                "Delta": observed_diff,
                "CI_L": np.percentile(boot_diffs, 2.5),
                "CI_U": np.percentile(boot_diffs, 97.5),
                "p": 2 * min(
                    np.mean(boot_diffs <= 0),
                    np.mean(boot_diffs >= 0),
                ),
                "B": boot_diffs.size,
            }
        )

    return pd.DataFrame(rows)


# ---------------------------------------------------------
# Using finalized recursive OOS predictions
# ---------------------------------------------------------

oos_main, results_main, oos_preds_main = run_recursive_experiment(
    panel=panel,
    target_col=TARGET_MAIN,
    features=FEATURES,
)

main_results = {
    "oos_raw": oos_main,
    "results_table": results_main,
    "oos_predictions": oos_preds_main,
}


# ---------------------------------------------------------
# Bootstrap AUC differences
# ---------------------------------------------------------

bootstrap_table = bootstrap_auc_table(
    oos_preds=main_results["oos_predictions"],
    target_col="y_true",
    model_map={
        "STRUCT": "p_struct",
        "LOGIT": "p_logit",
        "HYBRID": "p_hybrid_cv",
        "NAIVE": "p_naive",
        "LAG": "p_lag",
    },
    n_boot=1000,
    random_state=RANDOM_STATE,
)


bootstrap_table_display = bootstrap_table.copy()

if not bootstrap_table_display.empty:
    bootstrap_table_display[
        ["Delta", "CI_L", "CI_U", "p"]
    ] = bootstrap_table_display[
        ["Delta", "CI_L", "CI_U", "p"]
    ].round(3)

print("\nTable 6: Bootstrap AUC differences")
display(bootstrap_table_display)


# ---------------------------------------------------------
# Save outputs
# ---------------------------------------------------------

bootstrap_table_display.to_csv(
    os.path.join(
        TABLE_DIR,
        "bootstrap_auc_differences_main.csv",
    ),
    index=False,
)

print(
    "Saved bootstrap AUC differences table:",
    os.path.join(
        TABLE_DIR,
        "bootstrap_auc_differences_main.csv",
    ),
)


# ## APPENDICES: Supplementary Diagnostics and Technical Details

# #### APPENDIX TABLE A1. Discriminative component (ridge-logit): recursive coefficient estimates
# 

# In[16]:


# ---------------------------------------------------------
# One coefficient estimate per recursive forecast year
# ---------------------------------------------------------

oos_years_A1 = (
    oos_dates[["year"]]
    .drop_duplicates()
    .sort_values("year")
    .reset_index(drop=True)
)

# ---------------------------------------------------------
# Recursive coefficient estimation
# ---------------------------------------------------------

coef_df_A1 = expanding_logit_coefficients(
    df=panel,
    oos_dates=oos_years_A1,
    features=FEATURES,
    target_col=TARGET_MAIN,
)

# ---------------------------------------------------------
# Coefficient summary
# ---------------------------------------------------------

coef_summary_A1 = summarize_coefficients(
    coef_df_A1,
    FEATURES,
)


variable_labels = {
    "EAR_lag1": r"EAR$_{t-1}$",
    "ROA_lag1": r"ROA$_{t-1}$",
    "LLR_lag1": r"LLR$_{t-1}$",
    "LIQ_lag1": r"LIQ$_{t-1}$",
}

coef_summary_A1["Variable"] = (
    coef_summary_A1["Variable"]
    .map(variable_labels)
)

# ---------------------------------------------------------
# Sign column
# ---------------------------------------------------------

coef_summary_A1["Sign"] = np.where(
    coef_summary_A1["Mean"] >= 0,
    "+",
    "−",
)


coef_summary_A1 = coef_summary_A1[
    ["Variable", "Mean", "Std. Dev.", "Sign"]
]

# ---------------------------------------------------------
# Round values
# ---------------------------------------------------------

coef_summary_A1[["Mean", "Std. Dev."]] = (
    coef_summary_A1[["Mean", "Std. Dev."]]
    .round(3)
)

# ---------------------------------------------------------
# Display
# ---------------------------------------------------------

print("\nTable A1: Discriminative component (ridge-logit):")
print("recursive coefficient estimates")

display(coef_summary_A1)

# ---------------------------------------------------------
# Save outputs
# ---------------------------------------------------------

coef_df_A1.to_csv(
    os.path.join(
        TABLE_DIR,
        "appendix_recursive_logit_coefficients.csv",
    ),
    index=False,
)

coef_summary_A1.to_csv(
    os.path.join(
        TABLE_DIR,
        "table_A1_recursive_logit_coefficients.csv",
    ),
    index=False,
)

print("\nAppendix Table A1 saved successfully.")


# #### APPENDIX A2. CORRELATION, VIF, AND WOE COEFFICIENT DIAGNOSTICS

# In[17]:


from sklearn.linear_model import LogisticRegression
from statsmodels.stats.outliers_influence import variance_inflation_factor
import statsmodels.api as sm
import numpy as np
import pandas as pd


def compute_woe_map(x, y, n_bins=3, eps=0.5):
    """
    Estimate Weight-of-Evidence values for one continuous predictor.
    Bins are learned from the training sample only.
    """

    temp = pd.DataFrame(
        {
            "x": x,
            "y": y,
        }
    ).dropna().copy()

    if temp.empty or temp["y"].nunique() < 2:
        return None, None

    temp["bin"] = pd.qcut(
        temp["x"],
        q=n_bins,
        duplicates="drop",
    )

    total_good = (temp["y"] == 0).sum()
    total_bad = (temp["y"] == 1).sum()

    woe_map = {}

    for interval, group in temp.groupby("bin", observed=False):

        good = (group["y"] == 0).sum()
        bad = (group["y"] == 1).sum()

        dist_good = (good + eps) / (total_good + eps)
        dist_bad = (bad + eps) / (total_bad + eps)

        woe_map[interval] = np.log(dist_good / dist_bad)

    return woe_map, list(woe_map.keys())


def apply_woe(x, intervals, woe_map):
    """
    Apply an existing WoE mapping to a vector.
    Values outside the learned bin range are assigned to the nearest bin.
    """

    x = pd.Series(x).astype(float)

    values = []

    for value in x:

        if pd.isna(value):
            values.append(np.nan)
            continue

        matched_value = None

        for interval in intervals:
            if value in interval:
                matched_value = woe_map[interval]
                break

        if matched_value is None:
            if value <= intervals[0].left:
                matched_value = woe_map[intervals[0]]
            elif value >= intervals[-1].right:
                matched_value = woe_map[intervals[-1]]
            else:
                matched_value = np.nan

        values.append(matched_value)

    return np.asarray(values, dtype=float)


def save_diagnostics(
    panel,
    features=FEATURES,
    target_col=TARGET_MAIN,
    n_bins=3,
    random_state=RANDOM_STATE,
):
    """
    Save and return:
    Panel A: correlation matrix,
    Panel B: VIF diagnostics,
    Panel C: supplementary WoE coefficient diagnostics.
    """

    # -----------------------------------------------------
    # Panel A: Correlation matrix
    # -----------------------------------------------------

    diag_data = panel[features].dropna().reset_index(drop=True)

    if diag_data.empty:
        raise RuntimeError(
            "No complete rows available for correlation/VIF diagnostics."
        )

    corr_matrix = diag_data.corr()

    corr_matrix.to_csv(
        TABLE_DIR / "correlation_matrix.csv"
    )

    # -----------------------------------------------------
    # Panel B: VIF diagnostics
    # -----------------------------------------------------

    X_vif = sm.add_constant(diag_data)

    vif_table = pd.DataFrame(
        {
            "Variable": X_vif.columns,
            "VIF": [
                variance_inflation_factor(X_vif.values, i)
                for i in range(X_vif.shape[1])
            ],
        }
    )

    vif_table = (
        vif_table[vif_table["Variable"] != "const"]
        .reset_index(drop=True)
    )

    vif_table.to_csv(
        TABLE_DIR / "vif_diagnostics.csv",
        index=False,
    )

    # -----------------------------------------------------
    # Panel C: Recursive WoE coefficient diagnostics
    # -----------------------------------------------------

    panel_local = panel.sort_values(["country", "year"]).copy()

    panel_local["year"] = pd.to_numeric(
        panel_local["year"],
        errors="coerce",
    )

    panel_local = panel_local.dropna(subset=["year"]).copy()
    panel_local["year"] = panel_local["year"].astype(int)

    years = (
        panel_local["year"]
        .dropna()
        .astype(int)
        .sort_values()
        .unique()
    )

    oos_dates = years[WARMUP_YEARS:]

    coef_rows = []

    for h in oos_dates:

        train = panel_local.loc[
            panel_local["year"] < h,
            [target_col, *features],
        ].dropna().copy()

        if train.empty or train[target_col].nunique() < 2:
            continue

        y_train = train[target_col].astype(int).to_numpy()

        X_woe = pd.DataFrame(index=train.index)

        for feature in features:

            woe_map, intervals = compute_woe_map(
                train[feature],
                train[target_col],
                n_bins=n_bins,
            )

            if woe_map is None:
                X_woe[feature] = 0.0
                continue

            X_woe[feature] = apply_woe(
                train[feature],
                intervals,
                woe_map,
            )

        X_woe = (
            X_woe
            .replace([np.inf, -np.inf], np.nan)
            .fillna(0.0)
        )

        model = LogisticRegression(
            penalty="l2",
            C=1.0,
            solver="liblinear",
            random_state=random_state,
            max_iter=1000,
        )

        model.fit(X_woe, y_train)

        for feature, coef in zip(features, model.coef_[0]):
            coef_rows.append(
                {
                    "h": h,
                    "Variable": f"{feature} (WoE)",
                    "Coefficient": coef,
                }
            )

    woe_coef_draws = pd.DataFrame(coef_rows)

    if woe_coef_draws.empty:
        raise RuntimeError(
            "No WoE coefficients were estimated."
        )

    woe_coef_summary = (
        woe_coef_draws
        .groupby("Variable")
        .agg(
            Mean=("Coefficient", "mean"),
            **{"Std. Dev.": ("Coefficient", "std")}
        )
        .reset_index()
    )

    woe_coef_summary["Sign"] = np.where(
        woe_coef_summary["Mean"] < 0,
        "$-$",
        "$+$",
    )

    woe_coef_summary.to_csv(
        TABLE_DIR / "woe_coefficient_diagnostics.csv",
        index=False,
    )

    woe_coef_draws.to_csv(
        TABLE_DIR / "woe_coefficient_draws.csv",
        index=False,
    )

    return corr_matrix, vif_table, woe_coef_summary, woe_coef_draws


# ---------------------------------------------------------
# Run diagnostics
# ---------------------------------------------------------

corr_matrix, vif_table, woe_coef_summary, woe_coef_draws = save_diagnostics(
    panel=panel,
    features=FEATURES,
    target_col=TARGET_MAIN,
    n_bins=3,
    random_state=RANDOM_STATE,
)


# ---------------------------------------------------------
# Display outputs
# ---------------------------------------------------------

print("Panel A: Correlation matrix")
display(corr_matrix.round(3))

print("Panel B: VIF diagnostics")
display(vif_table.round(3))

print("Panel C: Supplementary WoE coefficient diagnostics")

woe_coef_summary_display = woe_coef_summary.copy()

woe_coef_summary_display[["Mean", "Std. Dev."]] = (
    woe_coef_summary_display[["Mean", "Std. Dev."]]
    .round(3)
)

display(woe_coef_summary_display)


# ---------------------------------------------------------
# Save display version
# ---------------------------------------------------------

woe_coef_summary_display.to_csv(
    TABLE_DIR / "woe_coefficient_diagnostics_display.csv",
    index=False,
)

print("Saved diagnostics successfully.")


# #### APPENDIX TABLE A3. Descriptive statistics of core variables

# In[18]:


DESC_VARS = {
    "EAR": "EAR",
    "ROA": "ROA",
    "LLR": "LLR",
    "LIQ": "LIQ",
    "WFS": "WFS",
    "stress_score": "Stress Score",
    TARGET_MAIN: "Base Distress",
    TARGET_EARLY_FRAGILITY: "Early Fragility",
}

# ---------------------------------------------------------
# Computing descriptive statistics
# ---------------------------------------------------------

desc_rows = []

for col, label in DESC_VARS.items():

    if col not in panel.columns:
        print(f"Skipping missing column: {col}")
        continue

    x = panel[col].dropna()

    desc_rows.append({
        "Variable": label,
        "Mean": x.mean(),
        "Std. Dev.": x.std(),
        "Min": x.min(),
        "Max": x.max(),
        "N": x.shape[0],
    })

table_A3 = pd.DataFrame(desc_rows)


table_A3[
    ["Mean", "Std. Dev.", "Min", "Max"]
] = (
    table_A3[
        ["Mean", "Std. Dev.", "Min", "Max"]
    ].round(3)
)

table_A3["N"] = table_A3["N"].astype(int)

print("\nTable A3: Summary statistics of core variables")
display(table_A3)

# ---------------------------------------------------------
# Saving outputs
# ---------------------------------------------------------

table_A3.to_csv(
    os.path.join(
        TABLE_DIR,
        "table_A3_descriptive_statistics.csv",
    ),
    index=False,
)

print("\nAppendix Table A3 saved successfully.")


# #### APPENDIX TABLE A4. Country-level distress frequencies

# In[19]:


# ---------------------------------------------------------
# Country-level aggregation
# ---------------------------------------------------------

country_rows = []

for country, g in panel.groupby("country"):

    country_rows.append({
        "Country": country,
        "N": g.shape[0],

        "Mean Score": g["stress_score"].mean(),

        "Base Distress Rate": g[TARGET_MAIN].mean(),

        "Alt. Distress Rate": g[
            TARGET_EARLY_FRAGILITY
        ].mean(),
    })

table_A4 = pd.DataFrame(country_rows)

# ---------------------------------------------------------
# Sort by country name
# ---------------------------------------------------------

table_A4 = (
    table_A4
    .sort_values("Country")
    .reset_index(drop=True)
)


table_A4[
    [
        "Mean Score",
        "Base Distress Rate",
        "Alt. Distress Rate",
    ]
] = (
    table_A4[
        [
            "Mean Score",
            "Base Distress Rate",
            "Alt. Distress Rate",
        ]
    ]
    .round(3)
)

table_A4["N"] = table_A4["N"].astype(int)

print("\nTable A4: Country-level distress frequencies")
display(table_A4)

# ---------------------------------------------------------
# Saving outputs
# ---------------------------------------------------------

table_A4.to_csv(
    os.path.join(
        TABLE_DIR,
        "table_A4_country_level_distress_frequencies.csv",
    ),
    index=False,
)

print("\nAppendix Table A4 saved successfully.")


# #### APPENDIX TABLE A5. Distribution of severe-distress events in the recursive out-of-sample panel

# In[20]:


# ---------------------------------------------------------
# Use main recursive OOS predictions
# ---------------------------------------------------------

if "main_results" in globals():
    oos_main_A5 = main_results["oos_predictions"].copy()
elif "oos_preds" in globals():
    oos_main_A5 = oos_preds.copy()
else:
    _, _, oos_main_A5 = run_recursive_experiment(
        panel=panel,
        target_col=TARGET_MAIN,
        features=FEATURES,
    )

# ---------------------------------------------------------
# Country-level OOS event distribution
# ---------------------------------------------------------

table_A5 = (
    oos_main_A5
    .groupby("country")
    .agg(
        N=(TARGET_MAIN, "size"),
        Severe_Events=(TARGET_MAIN, "sum"),
        Prevalence=(TARGET_MAIN, "mean"),
    )
    .reset_index()
    .rename(columns={
        "country": "Country",
        "Severe_Events": "Severe Events",
    })
)

# ---------------------------------------------------------
# Add total row
# ---------------------------------------------------------

total_row = pd.DataFrame({
    "Country": ["Total"],
    "N": [table_A5["N"].sum()],
    "Severe Events": [table_A5["Severe Events"].sum()],
    "Prevalence": [
        table_A5["Severe Events"].sum() / table_A5["N"].sum()
    ],
})

table_A5 = pd.concat(
    [table_A5, total_row],
    ignore_index=True,
)

# ---------------------------------------------------------
# Round and format
# ---------------------------------------------------------

table_A5["N"] = table_A5["N"].astype(int)
table_A5["Severe Events"] = table_A5["Severe Events"].astype(int)
table_A5["Prevalence"] = table_A5["Prevalence"].round(3)

print("\nTable A5: Distribution of severe-distress events")
print("in the recursive out-of-sample panel")

display(table_A5)

# ---------------------------------------------------------
# Save outputs
# ---------------------------------------------------------

table_A5.to_csv(
    os.path.join(
        TABLE_DIR,
        "table_A5_oos_severe_distress_distribution.csv",
    ),
    index=False,
)

print("\nAppendix Table A5 saved successfully.")


# #### APPENDIX A6. Persistence-overlap diagnostic

# In[21]:


panel_overlap = (
    panel
    .sort_values(["country", "year"])
    .copy()
    .reset_index(drop=True)
)

panel_overlap["prev_severe"] = (
    panel_overlap
    .groupby("country")[TARGET_MAIN]
    .shift(1)
)

panel_overlap["prev_early"] = (
    panel_overlap
    .groupby("country")[TARGET_EARLY_FRAGILITY]
    .shift(1)
)

def overlap_summary(df, target_col, prev_col, label):
    event_df = df[df[target_col] == 1].copy()

    total_events = len(event_df)
    continuing_events = int((event_df[prev_col] == 1).sum())
    onset_events = total_events - continuing_events

    overlap_rate = (
        continuing_events / total_events
        if total_events > 0
        else np.nan
    )

    return {
        "Environment": label,
        "Total Events": total_events,
        "Onset Events": onset_events,
        "Continuing Events": continuing_events,
        "Overlap Rate": overlap_rate,
    }

table_overlap = pd.DataFrame([
    overlap_summary(
        panel_overlap,
        TARGET_MAIN,
        "prev_severe",
        "Severe distress",
    ),
    overlap_summary(
        panel_overlap,
        TARGET_EARLY_FRAGILITY,
        "prev_early",
        "Early fragility",
    ),
])

table_overlap["Overlap Rate"] = (
    table_overlap["Overlap Rate"]
    .round(3)
)

print("\nPersistence-overlap diagnostic")
display(table_overlap)

table_overlap.to_csv(
    os.path.join(
        TABLE_DIR,
        "appendix_persistence_overlap_diagnostic.csv",
    ),
    index=False,
)

print("\nPersistence-overlap diagnostic saved.")


# #### Table A7: Leave-one-country-out influence analysis

# In[22]:


loo_rows = []

countries = sorted(panel["country"].dropna().unique())

for country_out in countries:

    panel_loo = (
        panel[panel["country"] != country_out]
        .copy()
        .reset_index(drop=True)
    )

    oos_dates_loo = make_oos_dates(
        panel_loo,
        warmup=WARMUP_YEARS,
    )

    if oos_dates_loo.empty:
        continue

    try:
        _, results_loo, oos_preds_loo = run_recursive_experiment(
            panel=panel_loo,
            target_col=TARGET_MAIN,
            features=FEATURES,
        )

        results_keep = results_loo[
            results_loo["Model"].isin(["STRUCT", "HYBRID", "LOGIT"])
        ].copy()

        for _, row in results_keep.iterrows():
            loo_rows.append({
                "Country Excluded": country_out,
                "Model": row["Model"],
                "AUC": row["AUC"],
                "AP": row["AP"],
                "Brier": row["Brier"],
                "N": row["N"],
                "Events": int(oos_preds_loo[TARGET_MAIN].sum()),
            })

    except Exception as e:
        print(f"Skipping {country_out}: {e}")

table_loo = pd.DataFrame(loo_rows)

table_loo_display = table_loo.copy()

table_loo_display[["AUC", "AP", "Brier"]] = (
    table_loo_display[["AUC", "AP", "Brier"]]
    .round(3)
)

print("\nLeave-one-country-out influence analysis")
display(table_loo_display)

table_loo_display.to_csv(
    os.path.join(
        TABLE_DIR,
        "appendix_leave_one_country_out_influence.csv",
    ),
    index=False,
)

print("\nLeave-one-country-out influence analysis saved.")


# #### Table A8: Threshold sensitivity robustness: Stricter distress definition: S_it >= 3

# In[23]:


# ---------------------------------------------------------
# Construct stricter distress target
# ---------------------------------------------------------

panel["y_strict"] = (
    panel["stress_score"] >= 3
).astype(int)

TARGET_STRICT = "y_strict"

# ---------------------------------------------------------
# Rebuild lag target
# ---------------------------------------------------------

panel["prev_target_strict"] = (
    panel
    .groupby("country")[TARGET_STRICT]
    .shift(1)
)

# ---------------------------------------------------------
# OOS dates
# ---------------------------------------------------------

oos_dates_strict = make_oos_dates(
    panel,
    warmup=WARMUP_YEARS,
)

# ---------------------------------------------------------
# Structural model
# ---------------------------------------------------------

oos_struct_strict = structural_oos(
    panel,
    oos_dates_strict,
    target_col=TARGET_STRICT,
    alpha=STRUCT_ALPHA,
    k=STRUCT_K,
)

# ---------------------------------------------------------
# Logistic model
# ---------------------------------------------------------

oos_logit_strict = logit_oos_recursive_pooled(
    panel,
    oos_dates_strict,
    FEATURES,
    target_col=TARGET_STRICT,
)

# ---------------------------------------------------------
# Merge predictions
# ---------------------------------------------------------

oos_strict = (
    oos_struct_strict
    .merge(
        oos_logit_strict,
        on=["country", "year", "h"],
        how="left",
    )
    .sort_values(["country", "year"])
    .reset_index(drop=True)
)

# ---------------------------------------------------------
# Benchmarks
# ---------------------------------------------------------

panel_strict = panel.copy()

panel_strict["prev_target"] = (
    panel_strict
    .groupby("country")[TARGET_STRICT]
    .shift(1)
)

oos_strict["p_naive"] = [
    compute_naive_prob(
        panel_strict,
        c,
        y,
        target_col=TARGET_STRICT,
    )
    for c, y in zip(
        oos_strict["country"],
        oos_strict["year"],
    )
]

oos_strict["p_lag"] = [
    compute_lag_prob(
        panel_strict,
        c,
        y,
        target_col=TARGET_STRICT,
    )
    for c, y in zip(
        oos_strict["country"],
        oos_strict["year"],
    )
]

# ---------------------------------------------------------
# Hybrid model
# ---------------------------------------------------------

fit_stack_strict, oos_preds_strict = cross_fitted_hybrid(
    oos_strict,
    target_col=TARGET_STRICT,
)

print("\nThreshold sensitivity robustness completed.")
print(fit_stack_strict)

# ---------------------------------------------------------
# Evaluation table
# ---------------------------------------------------------

model_map_strict = {
    "LAG": "p_lag",
    "STRUCT": "p_struct",
    "HYBRID": "p_hybrid_cv",
    "LOGIT": "p_logit",
    "NAIVE": "p_naive",
}

table_strict = evaluate_prediction_table(
    df=oos_preds_strict,
    target_col=TARGET_STRICT,
    model_map=model_map_strict,
)

table_strict_display = table_strict.copy()

metric_cols = [
    "Brier",
    "LogLoss",
    "RMSE",
    "AUC",
    "AP",
]

table_strict_display[metric_cols] = (
    table_strict_display[metric_cols]
    .round(3)
)

print("\nTABLE A8: Threshold sensitivity robustness")
display(table_strict_display)

# ---------------------------------------------------------
# Sample diagnostics
# ---------------------------------------------------------

print("\nThreshold sensitivity sample check:")
print("OOS observations:", len(oos_preds_strict))
print("Strict distress events:", int(oos_preds_strict[TARGET_STRICT].sum()))

# ---------------------------------------------------------
# Save outputs
# ---------------------------------------------------------

oos_preds_strict.to_csv(
    os.path.join(
        PRED_DIR,
        "oos_predictions_strict_distress.csv",
    ),
    index=False,
)

table_strict_display.to_csv(
    os.path.join(
        TABLE_DIR,
        "table_threshold_sensitivity.csv",
    ),
    index=False,
)

print("\nThreshold sensitivity outputs saved.")


# #### TABLE A9: Robustness analysis with macro-financial proxy augmentation

# In[24]:


RAW_MACRO_FEATURES = [
    "Assets_Growth_(YoY)",
    "Loans_Growth_(YoY)",
    "Funding_Cost_Proxy",
    "Net_Interest_Spread_Proxy",
]

missing_macro = [
    c for c in RAW_MACRO_FEATURES
    if c not in panel.columns
]

if missing_macro:
    raise ValueError(f"Missing manuscript macro proxy columns: {missing_macro}")

print("Available raw macro proxies:")
print(RAW_MACRO_FEATURES)

# ---------------------------------------------------------
# Create lagged macro proxy variables
# ---------------------------------------------------------

panel = (
    panel
    .sort_values(["country", "year"])
    .reset_index(drop=True)
)

macro_lag_features = []

for col in RAW_MACRO_FEATURES:
    lag_col = f"{col}_lag1"

    panel[lag_col] = (
        panel
        .groupby("country")[col]
        .shift(1)
    )

    macro_lag_features.append(lag_col)

print("\nLagged macro proxy features:")
print(macro_lag_features)

# ---------------------------------------------------------
# Final macro-augmented feature set
# ---------------------------------------------------------

MACRO_FEATURES_FINAL = FEATURES + macro_lag_features

print("\nFinal macro-augmented feature set:")
print(MACRO_FEATURES_FINAL)

# =========================================================
# Baseline recursive predictions
# =========================================================

oos_dates_base = make_oos_dates(
    panel,
    warmup=WARMUP_YEARS,
)

oos_struct_base = structural_oos(
    panel,
    oos_dates_base,
    target_col=TARGET_MAIN,
    alpha=STRUCT_ALPHA,
    k=STRUCT_K,
)

oos_logit_base = logit_oos_recursive_pooled(
    panel,
    oos_dates_base,
    FEATURES,
    target_col=TARGET_MAIN,
)

oos_base = (
    oos_struct_base
    .merge(
        oos_logit_base,
        on=["country", "year", "h"],
        how="left",
    )
    .sort_values(["country", "year"])
    .reset_index(drop=True)
)

panel_base = panel.copy()
panel_base["prev_target"] = (
    panel_base
    .groupby("country")[TARGET_MAIN]
    .shift(1)
)

oos_base["p_naive"] = [
    compute_naive_prob(panel_base, c, y, target_col=TARGET_MAIN)
    for c, y in zip(oos_base["country"], oos_base["year"])
]

oos_base["p_lag"] = [
    compute_lag_prob(panel_base, c, y, target_col=TARGET_MAIN)
    for c, y in zip(oos_base["country"], oos_base["year"])
]

fit_stack_base, oos_preds_base = cross_fitted_hybrid(
    oos_base,
    target_col=TARGET_MAIN,
)

# =========================================================
# Macro-augmented recursive predictions
# =========================================================

oos_struct_macro = structural_oos(
    panel,
    oos_dates_base,
    target_col=TARGET_MAIN,
    alpha=STRUCT_ALPHA,
    k=STRUCT_K,
)

oos_logit_macro = logit_oos_recursive_pooled(
    panel,
    oos_dates_base,
    MACRO_FEATURES_FINAL,
    target_col=TARGET_MAIN,
)

oos_macro = (
    oos_struct_macro
    .merge(
        oos_logit_macro,
        on=["country", "year", "h"],
        how="left",
    )
    .sort_values(["country", "year"])
    .reset_index(drop=True)
)

oos_macro["p_naive"] = [
    compute_naive_prob(panel_base, c, y, target_col=TARGET_MAIN)
    for c, y in zip(oos_macro["country"], oos_macro["year"])
]

oos_macro["p_lag"] = [
    compute_lag_prob(panel_base, c, y, target_col=TARGET_MAIN)
    for c, y in zip(oos_macro["country"], oos_macro["year"])
]

fit_stack_macro, oos_preds_macro = cross_fitted_hybrid(
    oos_macro,
    target_col=TARGET_MAIN,
)

# =========================================================
# Evaluation
# =========================================================

macro_results = evaluate_prediction_table(
    df=oos_preds_macro,
    target_col=TARGET_MAIN,
    model_map={
        "STRUCT": "p_struct",
        "HYBRID": "p_hybrid_cv",
        "LOGIT": "p_logit",
        "NAIVE": "p_naive",
        "LAG": "p_lag",
    },
)

baseline_results = evaluate_prediction_table(
    df=oos_preds_base,
    target_col=TARGET_MAIN,
    model_map={
        "STRUCT": "p_struct",
        "HYBRID": "p_hybrid_cv",
        "LOGIT": "p_logit",
        "NAIVE": "p_naive",
        "LAG": "p_lag",
    },
)

print("\nMacro-augmented model results")
display(macro_results)



required_cols = ["Model", "AUC", "Brier", "LogLoss"]

table_A9 = (
    baseline_results[required_cols]
    .rename(columns={
        "AUC": "Baseline_AUC",
        "Brier": "Baseline_Brier",
        "LogLoss": "Baseline_LogLoss",
    })
    .merge(
        macro_results[required_cols]
        .rename(columns={
            "AUC": "Macro_AUC",
            "Brier": "Macro_Brier",
            "LogLoss": "Macro_LogLoss",
        }),
        on="Model",
        how="inner",
    )
)

table_A9 = table_A9[
    table_A9["Model"].isin(["STRUCT", "HYBRID", "LOGIT"])
].copy()

table_A9["Model"] = pd.Categorical(
    table_A9["Model"],
    categories=["STRUCT", "HYBRID", "LOGIT"],
    ordered=True,
)

table_A9 = (
    table_A9
    .sort_values("Model")
    .reset_index(drop=True)
)

num_cols = table_A9.columns.drop("Model")
table_A9[num_cols] = table_A9[num_cols].round(3)

print("\nTABLE A9: Robustness analysis with macro-financial proxy augmentation")
display(table_A9)

# =========================================================
# Save outputs
# =========================================================

oos_preds_macro.to_csv(
    os.path.join(
        PRED_DIR,
        "oos_predictions_macro_augmented.csv",
    ),
    index=False,
)

macro_results.to_csv(
    os.path.join(
        TABLE_DIR,
        "macro_augmented_model_results.csv",
    ),
    index=False,
)

table_A9.to_csv(
    os.path.join(
        TABLE_DIR,
        "table_A9_macro_financial_proxy_augmentation.csv",
    ),
    index=False,
)

print("\nMacro augmentation results and Table A9 saved successfully.")


# 
# #### SENSITIVITY ANALYSIS FOR STRUCTURAL SHRINKAGE PARAMETER
# 

# In[25]:


from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
    log_loss
)


def structural_kappa_sensitivity(
    df,
    oos_dates,
    target_col,
    kappa_values=(1, 2, 5),
    alpha=STRUCT_ALPHA
):
    """
    Evaluate the structural model for several values of the
    empirical-Bayes shrinkage parameter kappa.

    The existing recursive forecasting functions are not altered.
    """

    results = []

    for kappa in kappa_values:

        predictions = []

        for _, row in oos_dates.iterrows():

            country = row["country"]
            year = int(row["year"])

            test = df[
                (df["country"] == country) &
                (df["year"] == year)
            ]

            if test.empty:
                continue

            actual = test[target_col].iloc[0]

            if pd.isna(actual):
                continue

            probability = compute_structural_prob(
                df=df,
                country=country,
                year=year,
                target_col=target_col,
                alpha=alpha,
                k=kappa
            )

            predictions.append({
                "country": country,
                "year": year,
                "actual": int(actual),
                "p_struct": probability
            })

        predictions_df = pd.DataFrame(predictions)

        if predictions_df.empty:
            raise ValueError(
                f"No valid predictions were produced for kappa={kappa}."
            )

        y_true = predictions_df["actual"].astype(int)

        p_hat = predictions_df["p_struct"].clip(
            lower=1e-15,
            upper=1 - 1e-15
        )

        auc = (
            roc_auc_score(y_true, p_hat)
            if y_true.nunique() == 2
            else np.nan
        )

        results.append({
            "kappa": kappa,
            "AUC": auc,
            "AP": average_precision_score(y_true, p_hat),
            "Brier": brier_score_loss(y_true, p_hat),
            "LogLoss": log_loss(
                y_true,
                p_hat,
                labels=[0, 1]
            ),
            "N": len(y_true),
            "Events": int(y_true.sum())
        })

    return pd.DataFrame(results)


#### Table A10: Sensitivity of the structural model to the shrinkage parameter

# In[26]:


kappa_results = structural_kappa_sensitivity(
    df=panel,
    oos_dates=oos_dates,
    target_col=TARGET_MAIN,
    kappa_values=[1, 2, 5],
    alpha=STRUCT_ALPHA
)

print(
    kappa_results.to_string(
        index=False,
        float_format=lambda x: f"{x:.3f}"
    )
)


# In[27]:


# End of the analysis

